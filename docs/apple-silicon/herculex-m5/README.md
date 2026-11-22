# Evidence for "HerculeX on the M5 Pro" (`apple-silicon.md`)

Saved logs from the HerculeX benchmarks on the M5 Pro, copied unchanged except that home-directory paths are shortened to `~/`.
`round2_window.excerpt.log` holds five lines, verbatim, of a longer run log; the rest of that log is about other work.

| number | file |
|---|---|
| Stock 9f59e96 read the 404-piece unit in 70.04 s with its default 4 samplers | `stock-u55-run1.events.jsonl` (`kind: read`, end − start; `pieces`), `stock-u55-run1.meta.txt` (commit 9f59e96, `env=` empty) |
| The initial M5 reader commit (5b1576b) read the same unit in 59.60 s, also with 4 samplers: 70.04 / 59.60 = 1.175, printed 1.18×; one run each | `m5-u55-s4.events.jsonl`, `m5-u55-s4.meta.txt` (commit 5b1576b, `env=` empty) |
| The current reader (1e72245, one check; 5b1576b's maps have no saved comparison) against stock's float maps on the same captured inputs: pooled r 0.99997 / 0.99998, worst single map r 0.99897 / 0.99978, max abs diff 0.013 / 0.010 (the 404-piece unit / the 48 largest pieces of a 6,236-piece unit) | `round2_window.excerpt.log`, lines `reader u55` and `reader u44big` |
| Sampling read-ahead (`HERCULEX_READ_AHEAD`, on by default in the fork): 48 of 48 sampled captures byte-identical to stock's | `round2_window.excerpt.log`, line `sampler 48 pieces` |
| Pieces with `HERCULEX_DRAW_BYTES=10e9`: 30,188 files, 0 differing from stock's production pieces of the same unit (`record.json` aside) | `round2_window.excerpt.log`, line `pieces files` |

The `pieces rows-0000-0012` line also quotes a production stock time typed into the run script; no saved log backs it, so no pieces speedup is claimed.
The `dirty=` count in the `meta.txt` files counts `git status --short` lines; the checkouts held only untracked benchmark outputs.
