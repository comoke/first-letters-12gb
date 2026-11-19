"""Summarise an ink_run.sh output dir: per-map stats, percentile-stretched previews, stage times, GPU peak.
Usage (inside fl-claim-a7:final, which has numpy + PIL):  python ink_summary.py <outdir> <ink_run.log>
Writes <outdir>/summary/{summary.json, summary.md, *.png}.
"""
import glob, json, os, re, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
out, runlog = sys.argv[1], sys.argv[2]
S = os.path.join(out, "summary"); os.makedirs(S, exist_ok=True)
res = {"maps": {}, "stages": [], "gpu": {}}

maps = sorted(glob.glob(os.path.join(out, "fwd_*.npy")) + glob.glob(os.path.join(out, "fused", "fwd_*.npy")))
for p in maps:
    name = os.path.relpath(p, out).replace(os.sep, "_")[:-4]
    a = np.load(p, mmap_mode="r").astype(np.float32)
    fin = np.isfinite(a); v = a[fin]
    st = {"shape": list(a.shape), "finite_frac": float(fin.mean())}
    if v.size:
        q = np.percentile(v, [1, 50, 99, 99.9])
        st.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                  p1=float(q[0]), p50=float(q[1]), p99=float(q[2]), p999=float(q[3]))
        lo, hi = q[0], q[3] if q[3] > q[0] else q[0] + 1e-6
        img = np.where(fin, np.clip((a - lo) / (hi - lo), 0, 1), 0)
        im = Image.fromarray((img * 255).astype(np.uint8))
        f = max(1, int(np.ceil(max(a.shape) / 2400)))
        if f > 1: im = im.resize((a.shape[1] // f, a.shape[0] // f), Image.BILINEAR)
        im.save(os.path.join(S, name + ".png")); st["preview"] = name + ".png"; st["preview_downsample"] = f
    res["maps"][name] = st

ct = os.path.join(out, "ct20.png")
if os.path.exists(ct):
    im = Image.open(ct); f = max(1, int(np.ceil(max(im.size) / 2400)))
    im.resize((im.size[0] // f, im.size[1] // f), Image.BILINEAR).save(os.path.join(S, "ct20_small.png"))

last = open(runlog).read().split("--- restart")[-1]
for m in re.finditer(r"\[(\d+)s\] (\S+)(?: pass \d+)? exit=(\S+).*?\((\d+)s\)", last):
    res["stages"].append({"stage": m.group(2), "exit": m.group(3), "seconds": int(m.group(4))})
res["status"] = "INK_DONE" if "INK_DONE" in last else ("INK_FAIL" if "INK_FAIL" in last else "unknown")

g = os.path.join(out, "logs", "gpu.csv")
if os.path.exists(g):
    mem = [int(l.split(",")[1].split()[0]) for l in open(g) if l.count(",") >= 2]
    if mem: res["gpu"] = {"peak_mib_total_card": max(mem), "samples": len(mem)}

json.dump(res, open(os.path.join(S, "summary.json"), "w"), indent=1)
with open(os.path.join(S, "summary.md"), "w") as fh:
    fh.write(f"# ink summary: {out}\n\nstatus: {res['status']}  GPU peak (whole card): {res['gpu'].get('peak_mib_total_card')} MiB\n\n")
    fh.write("| stage | exit | min |\n|---|---|---|\n")
    for s in res["stages"]: fh.write(f"| {s['stage']} | {s['exit']} | {s['seconds']/60:.1f} |\n")
    fh.write("\n| map | finite | p50 | p99 | p99.9 | max |\n|---|---|---|---|---|---|\n")
    for k, s in res["maps"].items():
        fh.write(f"| {k} | {s['finite_frac']:.3f} | {s.get('p50', float('nan')):.4g} | {s.get('p99', float('nan')):.4g} | "
                 f"{s.get('p999', float('nan')):.4g} | {s.get('max', float('nan')):.4g} |\n")
print(open(os.path.join(S, "summary.md")).read())
