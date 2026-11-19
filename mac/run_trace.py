"""TRACE v3B on MPS with the winner's ink_v2.py (villa VillaFast, tta8, depths -1/0/+1), unchanged except for device/precision.
Usage: python run_trace.py RENDER OUT --models tr --crop y0,y1,x0,x1 [--batch 8]   (ink_v2.py arguments)
Env MAC_AMP = ref | fp32 | fp16 | bf16.
"""
import os, sys, runpy, time, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mps_shim as S                                                          # noqa: E402
t = time.time()
sys.argv = ['ink_v2.py'] + sys.argv[1:]
try:
    runpy.run_path(f'{os.environ["FL_ROOT"]}/claim/ink_v2.py', run_name='__main__')
finally:
    print(json.dumps(dict(wall_s=round(time.time() - t, 1), mps_driver_gb=round(S.mem_gb()[1], 2),
                          max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2), amp=S.AMP)), flush=True)
