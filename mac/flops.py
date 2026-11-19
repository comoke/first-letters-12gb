"""MACs per tile of the v8-in network, per stage (conv layers only), via forward hooks on CPU meta shapes."""
import os, sys, json
sys.path.insert(0, os.environ['V8_DIR'])
import torch, torch.nn as nn
from ink8um import InkDetector
m = InkDetector.from_pretrained(os.environ['V8_DIR']).eval()
tot = {}; cur = ['pre']
def hook(name):
    def h(mod, i, o):
        k = mod.weight.shape; macs = o.numel() * k[1] * int(torch.tensor(k[2:]).prod())
        key = name.split('.')[1] if name.startswith('backbone.') else name.split('.')[0]
        tot[key] = tot.get(key, 0) + macs
    return h
for n, mod in m.named_modules():
    if isinstance(mod, (nn.Conv3d, nn.Conv2d)): mod.register_forward_hook(hook(n))
with torch.inference_mode(): m(torch.rand(1, 24, 64, 64))
T = sum(tot.values())
print(json.dumps({'GMAC_per_tile': round(T / 1e9, 1), 'split_pct': {k: round(100 * v / T, 1) for k, v in tot.items()}}))
