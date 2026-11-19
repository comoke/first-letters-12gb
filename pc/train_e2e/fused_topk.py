"""Fusion top-k agreement inside the claim box (plan item 8: top-1% Jaccard), our fused maps vs the shipped ones. Writes compare/fused_topk.json."""
import json, numpy as np
import os; DD = os.environ.get('VESUVIUS_DATA', '/mnt/d/Workspace/vesuvius-data')
O = DD + '/train_e2e/maps/fused/'; S = DD + '/winner/hf-data/results/fused/'
y0,y1,x0,x1=1639,2745,2431,6829; out={}
for k in ('prod','ensA','prodd'):
    a=np.load(O+f'fwd_{k}.npy',mmap_mode='r')[y0:y1,x0:x1].astype(np.float32); b=np.load(S+f'fwd_{k}.npy',mmap_mode='r')[y0:y1,x0:x1].astype(np.float32)
    m=np.isfinite(a)&np.isfinite(b); r={}
    for q in (1,5):
        ta=a>=np.percentile(a[m],100-q); tb=b>=np.percentile(b[m],100-q); ta&=m; tb&=m
        r[f'top{q}_jaccard']=round(float((ta&tb).sum()/(ta|tb).sum()),4)
    out[k]=dict(r, n=int(m.sum()), files=[O+f'fwd_{k}.npy', S+f'fwd_{k}.npy'])
out['box_render_px']=[y0,y1,x0,x1]; out['note']='fusion metric from the plan (item 8); computed after the chain, pc/train_e2e/fused_topk.py'
json.dump(out,open(DD + '/train_e2e/compare/fused_topk.json','w'),indent=1); print(json.dumps(out))
