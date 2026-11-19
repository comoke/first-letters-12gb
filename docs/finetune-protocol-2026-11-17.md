# Fine-tune demo protocol: the 12 GB recipe on new labels from Scroll 1 (PHercParis4), held-out test

**Status:** written and committed **before any GPU run** of this demo. The protocol changes only with a dated amendment below this line.

**Question:** can the winner's A-hn model, retrained with this repository's 12 GB recipe, adapt to a new scroll's labels? We measure on papyrus it never saw. A null is reported as a null.

## Data (Step 0 gates, all passed)
- **Volume:** the Vesuvius Challenge public Scroll 1 (PHercParis4) segment surface volumes, scan `20260411134726` (ESRF, 2.4 µm, 78 keV), level 2 (9.6 µm in-plane). Source: S3 `vesuvius-challenge-open-data`, `PHercParis4/segments/<id>/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/`.
- **Labels:** the organisers' ink labels aligned to that volume. HF bucket `scrollprize/datasets`, `ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-wNN/`: `_inklabels.zarr` and `_supervision_mask.zarr`, channel 10, 1 = ink.
  - These are hand strokes refined by iterative pseudo-labelling. No manually labelled Scroll 1 segment exists in the release (`man5` is a different scroll).
- **Train A = w03** (segment `20231106155351`). **Test B = w09** (segment `20230929220926`).
  - **Disjoint papyrus:** 0% of either segment's labelled points come within 50 voxels of the other surface, and their labelled areas are 3.9 mm apart in median radius. B is never used for any choice below.
- **Input recipe:**
  - source planes 7–103, averaged in groups of 4, give 24 layers centred on the labelled surface (plane 54);
  - the shipped 200 clip;
  - 9.6 µm, against the model's 8.64 µm: 1.11×, at the edge of its scale augmentation.
- **Packing:** the winner's `v8306lib.src_1447` and `make_packs.save_anchor`, unchanged. We write `inklabels.png`, `mask.png` and `layers/00..23.tif` per segment, and point `D1447` at our folder.

## Licence and what gets published
- The data is CC BY-NC 4.0 (S3 licence, volume metadata), with the EduceLab-Scrolls citation for Scroll 1.
- The registration terms that the ink-labels README links to (dl.ash2txt.org/LICENSE.txt) forbid publishing "ink label images" or textual revelations without the Vesuvius Challenge's written consent.
- **So the public write-up reports metrics only.** Images of the Scroll 1 maps or labels stay private unless the Vesuvius Challenge consents. We redistribute no data.

## Fixed before running
1. **Layer order** (uses A only). Run the shipped s0 zero-shot on A's labelled region in both layer orders, scored by AUC against A's labels. The higher one fixes the order for every arm.
   - **Kill:** if both orders give AUC < 0.55 on A, the inputs don't match the model. Stop and report.
2. **Arms.** All start from the shipped `hn4_s0` final checkpoint and use the 12 GB recipe (`grad_ckpt block`, `vram_gb 10.5`, `expandable_segments`). The shipped config is unchanged except for the items listed here.
   - **(a) Zero-shot:** the shipped s0, no training.
   - **(b) Fine-tuned on A:** the shipped mix (the five PHerc0306B regions and anchors w058, w060) plus A as a third anchor pack, uniform over anchor tiles. 300 steps. Two seeds, 1 and 2, because training isn't bit-reproducible on this card.
   - **(c) Control:** the same as (b) without A (anchors w058, w060 only), 300 steps, seeds 1 and 2. This separates "more training" from "new labels".
3. **Inference on B:** the A-hn recipe (`infer.py`, TTA 1) on B's labelled region with the chosen layer order. Same crop for every arm.
4. **Metrics on B, inside B's supervision mask:**
   - pixel AUC;
   - top-1% precision (the share of the top 1% of scores that are labelled ink);
   - F0.5 at a threshold of 0.5 on the model's probability.
5. **Success.** Both (b) seeds beat (a) and both (c) seeds on AUC by at least **0.02**. In each of those pairs, a paired bootstrap 95% interval of the AUC difference must exclude 0. The bootstrap uses 1,000 resamples of 512 px tiles (about 5 mm) of B's mask. Otherwise the result is reported as a null, or as "more training helps, new labels don't add", whichever the numbers show.
6. **Budget and cutoff:**
   - **GPU:** 6 h or less on the home RTX 4070, $0. Planned: layer-order check about 0.3 h, four 300-step runs about 1.7 h, five inferences on B about 2 h.
   - **Per run:** a run card in the log, and the host guard on.
   - **Cutoff:** results by **2026-11-25 23:59**, or the entry ships without the demo.
7. **Reporting:**
   - all arms and seeds, including failures;
   - caveats: model-refined labels, one scroll, one train/test pair, a different scanner and energy, and 9.6 vs 8.64 µm.

## Sources
- Gate evidence: the licence texts, segment geometry and sizes are in the local `finetune/gate/` folder of the data directory (`gates.md`).
- Winner code: `submission/code/training/x_push3/m306/train/v8306lib.py` (`src_1447`) and `.../self/code/make_packs.py` (`save_anchor`).

## Amendments
### Amendment 1 (2026-11-17, evening): a second, independent train/test pair. Written and committed before any GPU run of pair 2.

**Why:** pair 1 (A = w03, B = w09) met the success rule. One pair can't show whether that holds on other papyrus, so we repeat the identical experiment on a second pair that shares no papyrus with the first. **Pair 1's result and analysis are unchanged.** Both results are reported, whatever pair 2 shows.

**Pair 2: A2 = w06** (`20231210121321`, 1,007 mm² labelled) and **B2 = w00** (`20231016151002`, 553 mm² labelled).
- From the Step 0 geometry check (`finetune/gate/geo/geo_report.json`), 0.000 of labelled points lie within 50 voxels of the other surface, in both directions, for w06/w00, w06/w03, w06/w09, w00/w03 and w00/w09.
- So B2 shares no papyrus with any segment trained on in either pair, and A2 shares none with pair 1's test segment.
- w06 does touch w05 and w07, and w00 touches w01. None of those is used.
- Roles were fixed by cost, before any look at the data: the larger segment trains, and the smaller one keeps the five test maps cheap.

**Everything else is identical to the protocol above:**
- the volume, labels, input recipe and packing;
- the layer-order check and kill line, on A2 only: zero-shot s0 in both orders, kill if both AUC < 0.55, and the higher order is used for every arm;
- arms (a), (b1, b2) = s0 + A2 as third anchor pack and (c1, c2) = no A2, 300 steps, seeds 1 and 2;
- the metrics on B2 inside its supervision mask;
- the success rule (both b beat a and both c by ≥ 0.02 AUC, each paired 512 px tile bootstrap 95% CI excluding 0).

**Code:** the same scripts, made parametric without changing pair 1's defaults. `fetch_paris4.py` knows w06 and w00. `data_chain.sh` takes `TAGS`/`PACK`. `ft_chain.sh` takes `A_TAG=w06 B_TAG=w00 PFX=ft2 RES_NAME=results_p2`, and packs A2 after the order decision. Results go to `finetune/results_p2/`.

**Run card:**
- **Input:** S3 level-2 planes 7–102 of both segments and the `ink_9um` labels (sha256 of the packs logged).
- **Positive control:** the zero-shot order check on A2.
- **Control arms:** (c).
- **Expected GPU:**
  - order check about 1.0 h (w06's mask is about 1.8× w03's);
  - four training runs about 1.7 h;
  - five maps of B2 about 1.4 h;

  **≈ 4.1 h on the RTX 4070, $0.** That is more than the 3.3 h first estimated, because both orders are checked as the protocol requires.
- **Cap:** 6 h (the chain stops itself).
- **Cutoff:** unchanged.
- **Pre-emption:** a higher-priority GPU job on the same card pre-empts this chain. Each step is resumable, and a pause changes no result.

### Erratum (2026-11-18): planes 7–102, not 7–103
The input recipe above says "source planes 7–103". The code (`pc/finetune/fetch_paris4.py`: first plane 7, 96 planes) and Amendment 1 use planes 7–102, which pool into 24 groups of 4; 7–103 would be 97 planes. Both pairs ran with 7–102. Nothing else changes.
