"""ahn_coreml.py : convert the folded A-hn (ahn_fold.FoldedAhn) to a Core ML program and list what it is made of.

Conversion runs anywhere coremltools runs (Linux too, without a Core ML runtime); compiling and timing need the Mac.
Writes OUT.mlpackage (weights derived from CC BY-NC checkpoints: keep them local, never publish them) and OUT.ops.json:
op-type histogram, the rank of every tensor (the Neural Engine path wants rank <= 4), and each conv's shapes.
Usage: python ahn_coreml.py --ckpt .../hn4_s0_final/model.safetensors --out DIR/hn4_s0 [--mode stack|sum] [--batch 1]
       [--precision fp16|fp32] [--units ALL|CPU_AND_NE|CPU_AND_GPU]
Env: V8_DIR = the winner's third_party/v8in (ink8um).
"""
import argparse, collections, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ['V8_DIR'])
from ink8um import InkDetector                                                       # noqa: E402
import ahn_fold                                                                      # noqa: E402
import coremltools as ct                                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--mode', default='stack'); ap.add_argument('--batch', type=int, default=1)
ap.add_argument('--precision', default='fp16'); ap.add_argument('--units', default='ALL')
a = ap.parse_args()

cfg = json.load(open(os.path.join(os.environ['V8_DIR'], 'config.json')))
m = InkDetector(**cfg)
from safetensors.torch import load_file                                              # noqa: E402
m.load_state_dict(load_file(a.ckpt), strict=True)
net = ahn_fold.fold(m.eval(), a.mode).eval()
net.batch = a.batch                                                                  # fixed sizes for the trace
ex = torch.zeros(a.batch, 24, 64, 64)
t = time.time()
with torch.inference_mode():
    traced = torch.jit.trace(net, ex, strict=False)
mlm = ct.convert(traced, inputs=[ct.TensorType(name='tiles', shape=ex.shape, dtype=np.float32)],
                 outputs=[ct.TensorType(name='logits')], convert_to='mlprogram',
                 compute_precision=ct.precision.FLOAT16 if a.precision == 'fp16' else ct.precision.FLOAT32,
                 compute_units=getattr(ct.ComputeUnit, a.units), minimum_deployment_target=ct.target.macOS15)
conv_s = time.time() - t
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
mlm.save(a.out + '.mlpackage')

ops, ranks, convs = collections.Counter(), collections.Counter(), []


def walk(block):
    for op in block.operations:
        ops[op.op_type] += 1
        for o in op.outputs:
            if getattr(o, 'shape', None) is not None:
                ranks[len(o.shape)] += 1
        if op.op_type == 'conv':
            convs.append(dict(x=list(op.inputs['x'].shape), w=list(op.inputs['weight'].shape),
                              strides=[int(v) for v in op.inputs['strides'].val], out=list(op.outputs[0].shape)))
        for blk in getattr(op, 'blocks', []):
            walk(blk)


walk(mlm._mil_program.functions['main'])
rep = dict(mode=a.mode, batch=a.batch, precision=a.precision, units=a.units, convert_s=round(conv_s, 1),
           coremltools=ct.__version__, torch=torch.__version__, ops=dict(ops.most_common()), tensor_ranks=dict(ranks),
           convs=len(convs), conv_shapes=convs)
json.dump(rep, open(a.out + '.ops.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != 'conv_shapes'}, indent=1))
print('COREML_DONE', a.out + '.mlpackage', flush=True)
