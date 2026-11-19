# Stage 9: the claim image redrawn from our end-to-end maps, 2026-11-16

Driver: `pc/train_e2e/stage9.sh`, CPU only, inside the winner's image `fl-claim-a7:final` (its `run.sh final`). Logs: `docs/logs/stage9/` (`stage9.log` has the timings; `shipped.log` shows that the lead view `c6_grey` is `variants/TRACE_x_v8306H3_grey_page.png`, and the byte-for-byte MATCH).

## What ran

1. **The tool on the shipped maps** (`run.sh final`, no overrides): 296 s. It printed *"MATCH: byte-identical to the package PNGs"*, i.e. all 20 PNGs equal the sha256 values in the winner's package. So the drawing step itself is deterministic on this PC.
2. **The tool on our maps** (`MAPS=` override): 405 s. The map directory held:
   - from our end-to-end run: `fwd_prod`, `fwd_ensA`, `fwd_prodd` (stage 8 fusion of our retrained members) and `fwd_tr` (our exact TRACE reproduction, the map used in that fusion);
   - unchanged shipped maps for the models `train` mode does not retrain: `fwd_rv2`, `fwd_v3`, `fwd_v8`, `fwd_v8g3`, `fwd_v8q`.
   - CT layer and mesh: shipped.
3. **Pixel diff** of every variant PNG, ours vs the package's (`pc/train_e2e/stage9_diff.py` → `docs/train-e2e-stage9-diff-2026-11-16.json`). Differences are per pixel, max over the RGB channels, in 0–255 units. Images are 4417 × 7385 px.

## Result

**The lead claim image** (`c6_grey`: TRACE × A-hn, the package's `PHerc0306B_..._V5Rc.png`):

| measure | value |
|---|---|
| byte-identical | no |
| pixels that differ at all | 31.6% |
| pixels that differ by more than 8 / 255 | 13,580 (0.042%) |
| pixels that differ by more than 64 / 255 | 148 (0.00045%) |
| mean abs difference | 0.40 / 255 |
| max abs difference | 240 / 255 |

**Where the large differences are** (`pc/train_e2e/stage9_where.py` → `docs/train-e2e-stage9-where-2026-11-16.json`):
- All 148 pixels above 64 / 255 sit in one 25 × 21 px patch at the top of the image (rows 89–113, cols 4255–4275). That patch is a printed number in the image's caption: the display level, "z 7.96" in the package and "z 7.95" in ours (`docs/img/stage9-header-shipped-vs-ours.png`). The level is the 99.9th percentile of the fused map, so a 0.01 shift changes one digit.
- The 13,580 pixels above 8 / 255 form 202 blobs, the largest 1,476 px, spread over the image. They are small tone shifts where our fused map differs slightly from theirs (fused r 0.9992).
- A strip through the claim box, side by side, shows no visible difference (`docs/img/stage9-claimbox-strip-shipped-vs-ours.png`).

**Every other variant:**
- 11 of the 20 variants are byte-identical. None of them uses our retrained A-hn members; five include our TRACE map: `Rv2_TRACE_primary`, `Rv2_TRACE_secondary_rung_L050`, `Rv2_v8Q_grey_page`, `Rv2_v8Q_secondary_rung_L050`, `Rv2_v8in_primary`, `Rv2_v8in_secondary_rung_L050`, `TRACE_v8in_secondary_rung_L050`, `V3_TRACE_secondary_rung_L050`, `V3_grey_page`, `V3_v8Q_grey_page`, `v8in_grey_page`.
- Four more variants that use our TRACE map but no A-hn member differ by at most 1 / 255, in at most a few pixels (`TRACE_grey_page`, `Rv2_TRACE_grey_page`, `TRACE_v8in_grey_page`, `V3_TRACE_grey_page`). Our TRACE map is r 1.0000000 against the shipped one, with differences at the float16 step.
- Variants that use our retrained A-hn members differ as the lead does:
  - `TRACE_x_v8306H3_secondary_rung_L050` (threshold display): 1.46% of pixels differ, 0.028% by more than 8 / 255.
  - `TRACE_plus_v8306H3_grey_page`: 0.073% above 8 / 255, max 14.
  - `CONTROL_TRACE_x_DECOY_grey_page`: 0.0013% above 8 / 255.
  - `v8306H3_grey_page` (the A-hn mean alone, no TRACE): 3.3% above 8 / 255, mean 1.38 / 255. This is the variant most exposed to the retrain, as in the map comparison (ensA top-1% Jaccard 0.703).

## Reading

Redrawn from our 12 GB end-to-end maps, the winner's own claim-image step gives an image that differs from the package by more than 8 / 255 on 0.04% of its pixels, with the largest differences in one caption digit. The images that do not involve our retrained members come out byte-identical or within 1 / 255.

**Not claimed:** byte identity of the lead image, which retrained weights cannot give (GPU kernels are not bit-exact; the winner's `docs/REPRODUCE.md` says the same).
