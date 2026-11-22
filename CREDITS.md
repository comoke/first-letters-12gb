# Credits and licences

## This entry

| what | licence |
|---|---|
| our code and documentation (`pc/`, `mac/`, `docs/`, the top-level documents) | MIT ([`LICENSE`](LICENSE)) |
| maps, numbers, tables and images derived from the winner's or the Vesuvius Challenge data | CC BY-NC 4.0 ([`LICENSE-DATA`](LICENSE-DATA)) |

Everything we measured comes from the PHerc. 0306B First Letters winner's released model, data and maps, which in turn derive from the Vesuvius Challenge open data (CC BY-NC 4.0). Please credit the winner and the Vesuvius Challenge as well as us.

**No weights are shipped.** Our retrained checkpoints derive from CC BY-NC 4.0 data and from the CC BY-NC-SA 4.0 TRACE v3B weights, so this repository does not include them. Every script downloads the winner's own image, weights and data from their repositories (`pc/download_winner.sh`, `mac/fetch_winner.sh`).

## What we used

| item | used for | source | licence | by |
|---|---|---|---|---|
| PHerc. 0306B First Letters winning pipeline: code, Docker image `fl-claim-a7:final`, weights, training data, maps | everything | GitHub `SirProdigle/pherc0306b-first-letters`; HF `Prodigle/pherc0306b-first-letters`, `Prodigle/pherc0306b-ink-ahn` | code MIT; weights, data and maps CC BY-NC 4.0 | Liam Warner and Nicola Bodill |
| PHerc. 0306B scan, volume `20250521133212` (through the winner's claim render) | all CT | Vesuvius Challenge open data | CC BY-NC 4.0 | Vesuvius Challenge, ESRF |
| PHerc. 1447 surfaces w058 and w060 (the winner's anchor packs) | training | HF `YoussefMoNader/ink-8um-pherc1447-surfaces` | CC BY-NC 4.0 | Vesuvius Challenge; Youssef Nader et al. |
| v8-in | the A-hn members' starting weights; teacher | HF `YoussefMoNader/ink-8um-v8in` | MIT | Youssef Nader |
| TRACE v3B | half of the claim map; teacher | GitHub `Jashann/vesuvius-scrolling`, release v1.0 | code MIT; weights CC BY-NC-SA 4.0 | Jashann |
| Reader V3, Reader v2 | teachers | HF `domenicor046/reader-v3`, `domenicor046/reader-v2` | MIT | Domenico Russo (domenicor046) |
| 3-D convolutions run as 2-D kernels on Apple silicon (the idea behind the faster A-hn path) | the M5 Pro Core ML path | villa PR #2634 (open); HerculeX `stock_model.folded` | MIT | campbelltron1; Adam Betka |
| HerculeX, at 9f59e96 | the "HerculeX on the M5 Pro" note | GitHub `adambetka2/HerculeX` | MIT | Adam Betka (stokarz); Reader v3 inside it by Domenico Russo |
| VC3D (`vc_render_tifxyz`, build a4979b9, inside the winner's image) | renders | Vesuvius Challenge, GitHub `ScrollPrize/villa` | GPLv3 | Vesuvius Challenge |
| Scroll 1 (PHercParis4) segments w03 `20231106155351`, w09 `20230929220926`, w06 `20231210121321` and w00 `20231016151002`, scan `20260411134726`, level 2 | fine-tune demo (metrics, and two before/after crops published with the organisers' consent; no data redistributed) | Vesuvius Challenge open data (S3 `vesuvius-challenge-open-data`) | CC BY-NC 4.0 | Vesuvius Challenge, EduceLab |
| Scroll 1 aligned ink labels and supervision masks (`ink_9um`, `aligned-scrollprizeorg-21slices`) | fine-tune demo training (A) and test (B) | HF `scrollprize/datasets` | Vesuvius Challenge data terms; only the two consented crops published | Vesuvius Challenge |

This table follows the winner's own `docs/CREDITS.md`. Their image contains more third-party software, listed in their `submission/docker/THIRD_PARTY_NOTICES.md`.

**Scroll 1 citation.** The fine-tune demo uses Scroll 1 (PHercParis4) data. Please cite: Stephen Parsons, C. Seth Parker, Christy Chapman, Mami Hayashida and W. Brent Seales, "EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT", arXiv:2304.02084 (2023). The data is CC BY-NC 4.0 under the Vesuvius Challenge open-data terms; this repository reports metrics on it, plus two before/after crops published with the organisers' written consent (Paul, Vesuvius Challenge, 2026-11-20).
