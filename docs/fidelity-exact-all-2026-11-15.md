# Exact-recipe inference: every map of the claim, 4070 vs shipped, 2026-11-15

Addendum to `docs/fidelity-2026-11-13.md`. That document verified TRACE v3B and A-hn s0 with the exact shipped recipe; s1, s2, the decoy and the fused maps had been compared only on the default tile grid. This run adds them.

## Recipe

From the shipped sidecars `hf-data/results/member_maps/fwd_hn4_{s0,s1,s2,decoy}.json`; identical for all four members:
- roi 115 4166 694 7638, grid origin 115 694, `clip_to_roi` true;
- layer0 8, TTA 1, nsig 3, stride 21, min_cov 0.25.

Run with `pc/exact_s0.py <render> <out> <checkpoint>`, using each member's shipped checkpoint `hf-model/checkpoints/<m>_final/model.safetensors`, on `hf-data/results/claim_surface_render_41layers.zarr`.
- TRACE and s0 are the 2026-11-13 exact runs.
- Fusion: the winner's `fuse_v7.py`, with our TRACE, the four members and `--decoy`.
- Chain: `pc/exact_all_chain.sh`. Comparison: `pc/exact_all_compare.py`.
- Time on the RTX 4070: 3,446 s / 3,288 s / 3,285 s for s1 / s2 / decoy (source: `docs/logs/fidelity-exact/exact_all_chain.log`).

## Result

Source of every number below: `docs/fidelity-exact-all-2026-11-15.json` (printed table: `docs/logs/fidelity-exact/exact_all_compare.log`). Pixels compared = those finite in both maps.

| map (shipped file) | pixels | Pearson r | mean abs diff | max abs diff | share within 1e-3 |
|---|---|---|---|---|---|
| TRACE v3B (`member_maps/fwd_trace_v3B.npy`) | 28,134,195 | 1.0000000 | 7.9e-10 | 7.32e-4 | 100% |
| A-hn s0 (`member_maps/fwd_hn4_s0.npy`) | 27,526,276 | 0.9999999 | 5.34e-5 | 7.35e-4 | 100% |
| A-hn s1 (`member_maps/fwd_hn4_s1.npy`) | 27,526,276 | 0.9999999 | 5.06e-5 | 8.03e-4 | 100% |
| A-hn s2 (`member_maps/fwd_hn4_s2.npy`) | 27,526,276 | 0.9999999 | 5.56e-5 | 6.94e-4 | 100% |
| decoy (`member_maps/fwd_hn4_decoy.npy`) | 27,526,276 | 0.9999997 | 7.40e-5 | 1.02e-3 | 99.99999% (2 pixels above 1e-3) |
| A-hn mean (`fused/fwd_ensA.npy`) | 27,526,276 | 0.9999999 | 2.80e-5 | 4.88e-4 | 100% |
| product TRACE × A-hn (`fused/fwd_prod.npy`) | 27,520,437 | 0.9999999 | 6.74e-6 | 4.88e-4 | 100% |
| decoy product (`fused/fwd_prodd.npy`) | 27,520,437 | 0.9999998 | 1.71e-5 | 4.88e-4 | 100% |

Inside the claim box (render px 1639–2745 × 2431–6829): every map has r ≥ 0.9999997 and max abs diff ≤ 6.94e-4 (same JSON, key `claim_box`).

## Scope of the conclusion

On the RTX 4070, the winner's inference with the shipped recipe and checkpoints reproduces:
- every map the claim uses (TRACE v3B, the three A-hn members, the decoy member, the A-hn mean, the product and the decoy product);
- with r ≥ 0.9999997 and a maximum abs difference of 1.02e-3.

That maximum is the scale of the float16 storage step of the shipped maps (at values 0.5–1, one float16 step is 2.4e-4 to 4.9e-4). Only 2 decoy pixels exceed 1e-3.

**Not covered:**
- the depth-shuffle (`shuf_*`) maps: covered later, after we built the shuffled render with the winner's own script (`docs/train-e2e-results-2026-11-16.md`, "Follow-up runs"; `docs/train-e2e-compare/shuf_vs_shipped.json`);
- the claim image PNG redrawn from these maps (planned as train-e2e stage 9).
