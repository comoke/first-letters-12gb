"""Run the winner's CUDA-only inference code on Apple Silicon (MPS) without editing it.

The reference code hardcodes torch.autocast('cuda', ...). This shim, imported first, replaces torch.autocast with a wrapper
that sends the 'cuda' device type to the target device and applies the precision chosen by MAC_AMP:
  ref  : the reference precision (enabled when the caller asks for fp16/bf16; A-hn infer.py passes enabled=False off CUDA,
         so for 'ref' any non-fp32 dtype request turns autocast on)
  fp32 : autocast off (full fp32)
  fp16 / bf16 : autocast on with that dtype wherever the reference code autocasts at all
Env: MAC_DEVICE (default mps), MAC_AMP (default ref). Also sets INK_DEVICE for fastink.
"""
import os
import torch

DEVICE = os.environ.setdefault('MAC_DEVICE', 'mps')
os.environ['INK_DEVICE'] = DEVICE
AMP = os.environ.get('MAC_AMP', 'ref')
_DT = {'fp16': torch.float16, 'bf16': torch.bfloat16}
_orig = torch.amp.autocast_mode.autocast


def _autocast(device_type='cuda', dtype=None, enabled=True, cache_enabled=None):
    if device_type == 'cuda':
        device_type = DEVICE
    wants = dtype is not None and dtype != torch.float32
    if AMP == 'fp32':
        on, dt = False, None
    elif AMP == 'ref':
        on, dt = wants, dtype
    else:
        on, dt = wants, _DT[AMP]
    if device_type == 'cpu' and dt == torch.float16:
        dt = torch.bfloat16
    return _orig(device_type, dtype=dt, enabled=bool(on))


torch.autocast = _autocast


def sync():
    if DEVICE == 'mps':
        torch.mps.synchronize()


def mem_gb():
    """(current allocated, driver allocated) GB on MPS"""
    if DEVICE != 'mps':
        return 0.0, 0.0
    return torch.mps.current_allocated_memory() / 2 ** 30, torch.mps.driver_allocated_memory() / 2 ** 30
