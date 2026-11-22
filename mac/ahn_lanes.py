"""ahn_lanes.py : A-hn's Core ML program (ahn_coreml.py) on the Neural Engine and the GPU at the same time.

Each tile goes to one unit by a fixed pattern over the run's tile order (N the Neural Engine, G the GPU), so an uninterrupted rerun
sends every tile to the same unit. The calls are made from C (ahn_engine.m), outside Python's GIL, one thread per unit.
"""
import ctypes, hashlib, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = {'N': 3, 'G': 1}                                                             # MLComputeUnits: cpuAndNeuralEngine, cpuAndGPU


def _library(cache):
    src = os.path.join(HERE, 'ahn_engine.m')
    key = hashlib.sha256(open(src, 'rb').read()).hexdigest()[:16]
    lib = os.path.join(cache, f'ahn-engine-{key}.dylib')
    if not os.path.exists(lib):
        subprocess.run(['clang', '-O2', '-fobjc-arc', '-dynamiclib', '-framework', 'Foundation', '-framework', 'CoreML', '-o', lib, src],
                       check=True)
    calls = ctypes.CDLL(lib)
    calls.ahn_open.restype = ctypes.c_void_p
    calls.ahn_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    calls.ahn_predict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return calls


class Lanes:
    def __init__(self, mlpackage, pattern='GGNGN'):
        if not pattern or set(pattern) - set(UNITS):
            raise ValueError(f'lanes pattern {pattern!r}: letters N (Neural Engine) and G (GPU)')
        self.pattern, self.count = pattern, 0
        cache = os.path.dirname(os.path.abspath(mlpackage))
        self.calls = _library(cache)
        compiled = os.path.splitext(os.path.abspath(mlpackage))[0] + '.mlmodelc'
        self.engines = {u: self.calls.ahn_open(mlpackage.encode(), compiled.encode(), UNITS[u]) for u in sorted(set(pattern))}
        if not all(self.engines.values()):
            raise RuntimeError(f'could not open {mlpackage}')
        self.pools = {u: ThreadPoolExecutor(1) for u in self.engines}

    def _run(self, unit, tiles, out, picks):
        for i in picks:
            failed = self.calls.ahn_predict(ctypes.c_void_p(self.engines[unit]), tiles[i].ctypes.data, tiles.shape[1], tiles.shape[2],
                                            out[i].ctypes.data)
            if failed:
                raise RuntimeError(f'Core ML call failed on {unit}: {failed}')

    def __call__(self, tiles):
        """(B, 24, 64, 64) float32 -> (B, 1, 64, 64) float32 logits."""
        tiles = np.ascontiguousarray(tiles, dtype=np.float32)
        out = np.empty((len(tiles), 1, tiles.shape[2], tiles.shape[3]), dtype=np.float32)
        picks = {u: [] for u in self.engines}
        for i in range(len(tiles)):
            picks[self.pattern[(self.count + i) % len(self.pattern)]].append(i)
        self.count += len(tiles)
        for job in [self.pools[u].submit(self._run, u, tiles, out, p) for u, p in picks.items() if p]:
            job.result()
        return out
